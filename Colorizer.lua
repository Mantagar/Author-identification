local Colorizer={}

function Colorizer.red(text)
	return "\27[1;31m"..text.."\27[0m"
end

function Colorizer.green(text)
	return "\27[1;2;32m"..text.."\27[0m"
end

function Colorizer.yellow(text)
	return "\27[1;2;33m"..text.."\27[0m"
end

return Colorizer
